global class UCB_AccountAutomergeScheduler implements Schedulable
{    
    global void execute(SchedulableContext sc)
    {
        Id LogRecordTypeId = Schema.SObjectType.UCB_Log__c.getRecordTypeInfosByName().get('LSC Automerge Process').getRecordTypeId();
        UCB_Log__c log = new UCB_Log__c (
                                UCB_Status__c = 'In Progress',
                                UCB_Number_Of_Errors__c = 0,
                                UCB_Extended_Status__c = 'Summary of error Messages:',
                                UCB_Job_Items_Processed__c = 0,
                                UCB_Total_records_processed__c = 0,
                                RecordTypeId = LogRecordTypeId,
                                UCB_Job_Name__c = 'LSC Automerge Process',
                                UCB_User_Multi_Country_Code__c = 'ALL'
                            );
        Insert log;
        String LogId = log.id;
        system.debug('LogId' + LogId);
        
        Database.executeBatch(new UCB_AccountAutomergeBatch(LogId),10);
        // Database.executeBatch(AccountMergeBatch(LogId),10); 
    }
}